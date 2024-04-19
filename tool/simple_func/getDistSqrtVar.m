function distSqrtVar=getDistSqrtVar(pos,prob,patchNumPerPattern,partList,Name_batch,conf)
invalidNum=100000;

partNum=numel(partList);
truth=cell(partNum,1);
for i=1:partNum
    partID=partList(i);
    filename=sprintf('%spart_annotations_forEvaluation/truth_part%02d.mat',conf.data.readCode,partID);
    a=load(filename);
    truth{i}=a.truth;
end
[~,patNum,imgNum]=size(pos);
diff=zeros(2,partNum,patNum,imgNum);
for imgID=1:imgNum
    diff(:,:,:,imgID)=getDiff(pos(:,:,imgID),truth,imgID,invalidNum);
end
distSqrtVar=getAvgDistSqrtVar(diff,prob,patchNumPerPattern,invalidNum);
end


function diff=getDiff(pos,truth,imgID,invalidNum)
partNum=numel(truth);
patNum=size(pos,2);
diff=zeros(2,partNum,patNum);
for i=1:partNum
    pos_truth=repmat(truth{i}(imgID).pHW_center,[1,patNum]);
    if(isempty(pos_truth))
        diff(:,i,:)=invalidNum;
    else
        diff(:,i,:)=reshape(pos_truth-pos,[2,1,patNum]);
    end
end
end


function distSqrtVar=getAvgDistSqrtVar(diff,prob,patchNumPerPattern,invalidNum)
[~,partNum,patNum,~]=size(diff);
distSqrtVar=zeros(patNum,1);
for pat=1:patNum
    [~,list]=sort(prob(pat,:),'descend');
    list=list(1:patchNumPerPattern);
    for partID=1:partNum
        tmp=find(diff(1,partID,pat,:)==invalidNum);
        tmp=setdiff(list,tmp,'stable');
        dist=reshape(sqrt(sum(diff(:,partID,pat,tmp).^2,1)),[1,numel(tmp)]);
        distSqrtVar(pat)=distSqrtVar(pat)+sqrt(var(dist,[],2));
    end
    distSqrtVar(pat)=distSqrtVar(pat)/partNum;
end
end
